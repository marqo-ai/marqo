package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Test for sortBy feature in HybridSearcher.
 */
class SortByTest {

    /**
     * Test that verifies sorting of results based on a single sort field.
     */
    HitGroup helpGenerateHitGroupWithOnlySortFieldValue0() {
        FeatureData f1 = mock(FeatureData.class);
        when(f1.getDouble("sort_field_value_0")).thenReturn(-1e50);
        Hit doc1 = new Hit("doc1", 0.55);
        doc1.setField("matchfeatures", f1);

        FeatureData f2 = mock(FeatureData.class);
        when(f2.getDouble("sort_field_value_0")).thenReturn(1.0);
        Hit doc2 = new Hit("doc2", 0.65);
        doc2.setField("matchfeatures", f2);

        FeatureData f3 = mock(FeatureData.class);
        when(f3.getDouble("sort_field_value_0")).thenReturn(2.0);
        Hit doc3 = new Hit("doc3", 0.75);
        doc3.setField("matchfeatures", f3);

        FeatureData f4 = mock(FeatureData.class);
        when(f4.getDouble("sort_field_value_0")).thenReturn(2.0);
        Hit doc4 = new Hit("doc4", 0.85);
        doc4.setField("matchfeatures", f4);

        FeatureData f5 = mock(FeatureData.class);
        when(f5.getDouble("sort_field_value_0")).thenReturn(-1e50);
        Hit doc5 = new Hit("doc5", 0.45);
        doc5.setField("matchfeatures", f5);

        FeatureData f6 = mock(FeatureData.class);
        when(f6.getDouble("sort_field_value_0")).thenReturn(5.0);
        Hit doc6 = new Hit("doc6", 0.95);
        doc6.setField("matchfeatures", f6);

        FeatureData f7 = mock(FeatureData.class);
        when(f7.getDouble("sort_field_value_0")).thenReturn(6.0);
        Hit doc7 = new Hit("doc7", 0.90);
        doc7.setField("matchfeatures", f7);

        FeatureData f8 = mock(FeatureData.class);
        when(f8.getDouble("sort_field_value_0")).thenReturn(7.0);
        Hit doc8 = new Hit("doc8", 0.80);
        doc8.setField("matchfeatures", f8);

        FeatureData f9 = mock(FeatureData.class);
        when(f9.getDouble("sort_field_value_0")).thenReturn(8.0);
        Hit doc9 = new Hit("doc9", 0.70);
        doc9.setField("matchfeatures", f9);

        FeatureData f10 = mock(FeatureData.class);
        when(f10.getDouble("sort_field_value_0")).thenReturn(9.0);
        Hit doc10 = new Hit("doc10", 1.00);
        doc10.setField("matchfeatures", f10);

        HitGroup hits = new HitGroup();
        hits.add(doc1);
        hits.add(doc2);
        hits.add(doc3);
        hits.add(doc4);
        hits.add(doc5);
        hits.add(doc6);
        hits.add(doc7);
        hits.add(doc8);
        hits.add(doc9);
        hits.add(doc10);
        return hits;
    }

    /**
     * Test that verifies sorting of results based on a single sort field with ascending order
     * and last missing policy.
     */
    @Test
    void sort1FieldWithAscOrderAndLastMissingPolicy() {
        // build 10 distinct docs by hand

        HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
        HybridSearcher searcher = new HybridSearcher();
        String sortJson =
                "[{"
                        + "\"field_name\":\"ignored\","
                        + "\"order\":\"asc\","
                        + "\"missing\":\"last\""
                        + "}]";

        // full-depth, no trim
        // expected:
        // 1) doc2 (1.0)
        // 2) doc3 & doc4 both 2.0 → tie by original relevance: doc4(0.85) before doc3(0.75)
        // 3) doc6(5),doc7(6),doc8(7),doc9(8),doc10(9)
        // 4) missing last: doc1,doc5
        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc2", "doc4", "doc3", "doc6", "doc7", "doc8", "doc9", "doc10", "doc1",
                        "doc5");
    }

    /**
     * Test that verifies sorting of results based on a single sort field with desc order
     * and first missing policy.
     */
    @Test
    void sort1FieldWithDescOrderAndFirstMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
        HybridSearcher searcher = new HybridSearcher();
        String sortJson =
                "[{"
                        + "\"field_name\":\"ignored\","
                        + "\"order\":\"desc\","
                        + "\"missing\":\"first\""
                        + "}]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc1", // missing first  (–1e50, highest missing rel=0.55)
                        "doc5", // missing second (–1e50, next missing rel=0.45)
                        "doc10", // sort=9.0
                        "doc9", // sort=8.0
                        "doc8", // sort=7.0
                        "doc7", // sort=6.0
                        "doc6", // sort=5.0
                        "doc4", // sort=2.0, tie-break on original rel=0.85 (before doc3)
                        "doc3", // sort=2.0, tie-break rel=0.75
                        "doc2" // sort=1.0
                        );
    }

    /**
     * Helper that builds 6 docs with two independent sort_field values:
     *  - doc1/doc2 share field0=1.0 but doc2.field1=5.0 < doc1.field1=10.0
     *  - doc3/doc4 share field0=2.0 but doc3.field1=3.0 < doc4.field1=7.0
     *  - doc5/doc6 both missing (–1e50 → null)
     */
    HitGroup helpGenerateHitGroupWithTwoSortFieldValues() {
        HitGroup hits = new HitGroup();
        double[][] values = {
            {1.0, 10.0},
            {1.0, 5.0},
            {2.0, 3.0},
            {2.0, 7.0},
            {-1e50, -1e50},
            {-1e50, -1e50}
        };
        double[] relevances = {0.10, 0.20, 0.30, 0.40, 0.05, 0.06};
        for (int i = 0; i < values.length; i++) {
            FeatureData f = mock(FeatureData.class);
            when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
            when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
            Hit h = new Hit("doc" + (i + 1), relevances[i]);
            h.setField("matchfeatures", f);
            hits.add(h);
        }
        return hits;
    }

    @Test
    void sort2FieldsWithAscOrderAndLastMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithTwoSortFieldValues();
        HybridSearcher searcher = new HybridSearcher();
        // first sort_field_value_0 asc, missing last
        // then sort_field_value_1 asc, missing last
        String sortJson =
                "["
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"}"
                        + "]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        // Expected:
        // 1) doc2 (1.0,5.0) before doc1 (1.0,10.0)
        // 2) doc3 (2.0,3.0) before doc4 (2.0,7.0)
        // 3) missing last: doc6 (rel=0.06) before doc5 (rel=0.05)
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc2", "doc1",
                        "doc3", "doc4",
                        "doc6", "doc5");
    }

    /**
     * Helper that builds 4 docs with three sort_field values:
     *  - docA/B/C all have field0=1.0 but differ on field1/field2
     *  - docD missing all three
     */
    HitGroup helpGenerateHitGroupWithThreeSortFieldValues() {
        HitGroup hits = new HitGroup();
        String[] ids = {"docA", "docB", "docC", "docD"};
        double[][] values = {
            {1.0, 1.0, 3.0}, // docA
            {1.0, 1.0, 2.0}, // docB
            {1.0, 2.0, 1.0}, // docC
            {-1e50, -1e50, -1e50} // docD missing all
        };
        double[] relevances = {0.40, 0.50, 0.60, 0.70};
        for (int i = 0; i < ids.length; i++) {
            FeatureData f = mock(FeatureData.class);
            when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
            when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
            when(f.getDouble("sort_field_value_2")).thenReturn(values[i][2]);
            Hit h = new Hit(ids[i], relevances[i]);
            h.setField("matchfeatures", f);
            hits.add(h);
        }
        return hits;
    }

    @Test
    void sort3FieldsWithAscOrderAndFirstMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithThreeSortFieldValues();
        HybridSearcher searcher = new HybridSearcher();
        // all three ascending, missing first
        String sortJson =
                "[{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"}]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        // Expected:
        // 1) missing first: docD
        // 2) among the rest, field0 ties=1.0 → use field1:
        //      docA/B have field1=1.0 < docC.field1=2.0 → so docA & docB
        //    then break tie on field2: docB(2.0) < docA(3.0)
        // 3) then docC
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly("docD", "docB", "docA", "docC");
    }

    private Execution makeEmptyExec() {
        Execution exec = mock(Execution.class);
        when(exec.search(any(Query.class)))
                .thenAnswer(
                        invocation -> {
                            Query q = invocation.getArgument(0);
                            return new Result(q, new HitGroup());
                        });
        return exec;
    }

    /**
     * Test that verifies that postProcessBySort is called when sortBy is set in the query.
     */
    @Test
    void whenSortByFields_set_postProcessBySortIsCalled() {
        // 1) Create a Mockito spy on the real HybridSearcher
        HybridSearcher spy = spy(new HybridSearcher());

        // 2) Stub out createSubQuery (both overloads) so we never NPE inside it
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(any(Query.class), anyString(), anyString(), anyBoolean());
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(any(Query.class), anyString(), anyString(), anyBoolean());

        // 3) Stub extractTensorRankFeature to return:
        //    • null for "mult_weights_global" or "add_weights_global"
        //    • an empty Tensor for everything else (fields_to_rank_*)
        doAnswer(
                        inv -> {
                            String name = inv.getArgument(1);
                            if (name.contains("mult_weights_global")
                                    || name.contains("add_weights_global")) {
                                return null;
                            }
                            // non-null so createSubQuery and co. won't blow up
                            return Tensor.from("tensor(p{}):{}");
                        })
                .when(spy)
                .extractTensorRankFeature(any(Query.class), anyString());

        // 4) Stub postProcessBySort so it just returns an empty HitGroup
        doReturn(new HitGroup())
                .when(spy)
                .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());

        // 5) Build a Query that triggers the sortBy branch
        Query q = new Query("?q");
        q.properties().set("hits", 1);
        q.properties().set("offset", 0);
        q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
        q.properties().set("marqo__hybrid.rankingMethod", "lexical");
        q.properties()
                .set(
                        "marqo__hybrid.sortBy.fields",
                        "[{\"field_name\":\"foo\",\"order\":\"asc\",\"missing\":\"last\"}]");
        // MUST set this to avoid the NPE you saw
        q.properties().set("marqo__hybrid.sortBy.minSortCandidates", 10);

        // 6) Call search()
        spy.search(q, makeEmptyExec());

        // 7) Verify that only postProcessBySort() ran
        verify(spy, times(1))
                .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());
        verify(spy, never())
                .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());
    }

    /*
       Test that verifies that postProcessResults is called when only modifiers exist
       (i.e., no sortBy.fields).
    */
    @Test
    void whenOnlyModifiersExist_postProcessResultsIsCalled() {
        HybridSearcher spy = spy(new HybridSearcher());
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(any(), anyString(), anyString(), anyBoolean());
        // simulate "has a global mult modifier" but no sortBy
        Tensor dummy = Tensor.from("tensor<float>(d0[1]):[1]");
        doReturn(dummy).when(spy).extractTensorRankFeature(any(), contains("mult_weights_global"));
        doReturn(null).when(spy).extractTensorRankFeature(any(), contains("add_weights_global"));
        doReturn(new HitGroup())
                .when(spy)
                .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());

        Query q = new Query("?q");
        q.properties().set("hits", 1);
        q.properties().set("offset", 0);
        q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
        q.properties().set("marqo__hybrid.rankingMethod", "lexical");
        // no sortBy.fields

        spy.search(q, makeEmptyExec());

        verify(spy, times(1)).postProcessResults(any(), eq(q), any(), eq(1), eq(0), eq(false));
        verify(spy, never()).postProcessBySort(any(), anyString(), any(), anyInt(), anyInt());
    }

    @Nested
    class SortJsonParsingTest {

        private HitGroup createDummyHitGroup() {
            FeatureData f1 = mock(FeatureData.class);
            when(f1.getDouble("sort_field_value_0")).thenReturn(1.0);
            when(f1.getDouble("sort_field_value_1")).thenReturn(2.0);
            when(f1.getDouble("sort_field_value_2")).thenReturn(3.0);
            Hit doc1 = new Hit("doc1", 0.5);
            doc1.setField("matchfeatures", f1);

            HitGroup hits = new HitGroup();
            hits.add(doc1);

            return hits;
        }

        @Test
        void shouldParseValidSortJsonWith1Field() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String sortJson = "[{\"field_name\":\"score\",\"order\":\"asc\",\"missing\":\"last\"}]";

            // Should not throw exception
            HitGroup result = searcher.postProcessBySort(hits, sortJson, null, 10, 0);
            assertThat((Object) result).isNotNull();
            assertThat(result.asList()).hasSize(1);
        }

        @Test
        void shouldParseValidSortJsonWith2Fields() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String sortJson =
                    "[{\"field_name\":\"score\",\"order\":\"desc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"timestamp\",\"order\":\"asc\",\"missing\":\"last\"}]";

            // Should not throw exception
            HitGroup result = searcher.postProcessBySort(hits, sortJson, null, 10, 0);
            assertThat((Object) result).isNotNull();
            assertThat(result.asList()).hasSize(1);
        }

        @Test
        void shouldParseValidSortJsonWith3Fields() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String sortJson =
                    "[{\"field_name\":\"score\",\"order\":\"desc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"timestamp\",\"order\":\"asc\",\"missing\":\"last\"},"
                        + "{\"field_name\":\"category\",\"order\":\"desc\",\"missing\":\"first\"}]";

            // Should not throw exception
            HitGroup result = searcher.postProcessBySort(hits, sortJson, null, 10, 0);
            assertThat((Object) result).isNotNull();
            assertThat(result.asList()).hasSize(1);
        }

        @Test
        void shouldParseVariousMissingAndOrderValues() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test different combinations of order and missing values
            String[] testCases = {
                "[{\"field_name\":\"f1\",\"order\":\"ASC\",\"missing\":\"FIRST\"}]",
                "[{\"field_name\":\"f1\",\"order\":\"DESC\",\"missing\":\"LAST\"}]",
                "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"first\"}]",
                "[{\"field_name\":\"f1\",\"order\":\"desc\",\"missing\":\"last\"}]",
                "[{\"field_name\":\"f1\",\"order\":\"Asc\",\"missing\":\"First\"}]",
                "[{\"field_name\":\"f1\",\"order\":\"Desc\",\"missing\":\"Last\"}]"
            };

            for (String sortJson : testCases) {
                // Should not throw exception for any of these cases
                HitGroup result = searcher.postProcessBySort(hits, sortJson, null, 10, 0);
                assertThat((Object) result).isNotNull();
            }
        }

        @Test
        void shouldThrowExceptionForNullOrderAndMissing() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test with null order value
            String sortJsonNullOrder =
                    "[{\"field_name\":\"f1\",\"order\":null,\"missing\":\"last\"}]";
            assertThatThrownBy(
                            () -> searcher.postProcessBySort(hits, sortJsonNullOrder, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("order is required for sort field at index 0");

            // Test with null missing value
            String sortJsonNullMissing =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":null}]";
            assertThatThrownBy(
                            () ->
                                    searcher.postProcessBySort(
                                            hits, sortJsonNullMissing, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("missing is required for sort field at index 0");
        }

        @Test
        void shouldThrowExceptionForInvalidJson() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String invalidJson = "{invalid json}";

            assertThatThrownBy(() -> searcher.postProcessBySort(hits, invalidJson, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "Invalid sort JSON format for marqo__hybrid.sortBy.fields");
        }

        @Test
        void shouldThrowExceptionForMalformedJsonArray() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String malformedJson =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\",}]"; // trailing
            // comma

            assertThatThrownBy(() -> searcher.postProcessBySort(hits, malformedJson, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "Invalid sort JSON format for marqo__hybrid.sortBy.fields");
        }

        @Test
        void shouldThrowExceptionForEmptyArray() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String emptySortJson = "[]";

            // Should throw exception for empty sort fields
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, emptySortJson, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBy fields cannot be empty. Must contain 1 to 3 sort fields.");
        }

        @Test
        void shouldThrowExceptionForTooManyFields() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Build JSON with more than 3 fields (4 fields)
            String sortJsonWith4Fields =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"order\":\"desc\",\"missing\":\"first\"},"
                            + "{\"field_name\":\"f3\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f4\",\"order\":\"desc\",\"missing\":\"first\"}]";

            // Should throw exception for more than 3 fields
            assertThatThrownBy(
                            () ->
                                    searcher.postProcessBySort(
                                            hits, sortJsonWith4Fields, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBy fields cannot contain more than 3 sort fields. Found: 4");

            // Build JSON with 5 fields to test the counter
            String sortJsonWith5Fields =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"order\":\"desc\",\"missing\":\"first\"},"
                            + "{\"field_name\":\"f3\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f4\",\"order\":\"desc\",\"missing\":\"first\"},"
                            + "{\"field_name\":\"f5\",\"order\":\"asc\",\"missing\":\"last\"}]";

            assertThatThrownBy(
                            () ->
                                    searcher.postProcessBySort(
                                            hits, sortJsonWith5Fields, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBy fields cannot contain more than 3 sort fields. Found: 5");
        }

        @Test
        void shouldThrowExceptionForUnsupportedSortOrderValues() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            String[] unsupportedOrders = {"ascending", "descending", "up", "down", "invalid", ""};

            for (String order : unsupportedOrders) {
                String sortJson =
                        "[{\"field_name\":\"f1\",\"order\":\""
                                + order
                                + "\",\"missing\":\"last\"}]";
                // Should throw exception for unsupported order values
                assertThatThrownBy(() -> searcher.postProcessBySort(hits, sortJson, null, 10, 0))
                        .isInstanceOf(RuntimeException.class)
                        .hasMessageContaining(
                                "Invalid sort JSON format for marqo__hybrid.sortBy.fields");
            }
        }

        @Test
        void shouldThrowExceptionsForUnsupportedMissingValues() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test unsupported missing values - should default to FIRST
            String[] unsupportedMissing = {"top", "bottom", "start", "end", "invalid", ""};

            for (String missing : unsupportedMissing) {
                String sortJson =
                        "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\""
                                + missing
                                + "\"}]";
                // Should throw exception for unsupported missing values
                assertThatThrownBy(() -> searcher.postProcessBySort(hits, sortJson, null, 10, 0))
                        .isInstanceOf(RuntimeException.class)
                        .hasMessageContaining(
                                "Invalid sort JSON format for marqo__hybrid.sortBy.fields");
            }
        }

        @Test
        void shouldThrowExceptionForMissingRequiredFields() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test JSON missing field_name
            String missingFieldName = "[{\"order\":\"asc\",\"missing\":\"last\"}]";
            assertThatThrownBy(
                            () -> searcher.postProcessBySort(hits, missingFieldName, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("fieldName is required for sort field at index 0");

            // Test JSON missing order
            String missingOrder = "[{\"field_name\":\"f1\",\"missing\":\"last\"}]";
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, missingOrder, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("order is required for sort field at index 0");

            // Test JSON missing missing
            String missingMissing = "[{\"field_name\":\"f1\",\"order\":\"asc\"}]";
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, missingMissing, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("missing is required for sort field at index 0");

            // Test empty field_name
            String emptyFieldName =
                    "[{\"field_name\":\"\",\"order\":\"asc\",\"missing\":\"last\"}]";
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, emptyFieldName, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("fieldName is required for sort field at index 0");

            // Test whitespace-only field_name
            String whitespaceFieldName =
                    "[{\"field_name\":\"   \",\"order\":\"asc\",\"missing\":\"last\"}]";
            assertThatThrownBy(
                            () ->
                                    searcher.postProcessBySort(
                                            hits, whitespaceFieldName, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("fieldName is required for sort field at index 0");
        }

        @Test
        void shouldAcceptValidSortFieldCounts() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test with exactly 1 field (should work)
            String oneField = "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"}]";
            HitGroup result1 = searcher.postProcessBySort(hits, oneField, null, 10, 0);
            assertThat((Object) result1).isNotNull();
            assertThat(result1.asList()).hasSize(1);

            // Test with exactly 2 fields (should work)
            String twoFields =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"order\":\"desc\",\"missing\":\"first\"}]";
            HitGroup result2 = searcher.postProcessBySort(hits, twoFields, null, 10, 0);
            assertThat((Object) result2).isNotNull();
            assertThat(result2.asList()).hasSize(1);

            // Test with exactly 3 fields (should work - boundary case)
            String threeFields =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"order\":\"desc\",\"missing\":\"first\"},"
                            + "{\"field_name\":\"f3\",\"order\":\"asc\",\"missing\":\"last\"}]";
            HitGroup result3 = searcher.postProcessBySort(hits, threeFields, null, 10, 0);
            assertThat((Object) result3).isNotNull();
            assertThat(result3.asList()).hasSize(1);
        }

        @Test
        void shouldThrowWhenSortFieldJsonIsMissingOrderOrFieldNameAtCorrectIndex() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();

            // Test that validation works for fields at different indices
            String invalidSecondField =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"missing\":\"first\"}]"; // missing order in
            // 2nd field

            assertThatThrownBy(
                            () -> searcher.postProcessBySort(hits, invalidSecondField, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("order is required for sort field at index 1");

            String invalidThirdField =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"},"
                            + "{\"field_name\":\"f2\",\"order\":\"desc\",\"missing\":\"first\"},"
                            + "{\"order\":\"asc\",\"missing\":\"last\"}]"; // missing field_name in
            // 3rd field

            assertThatThrownBy(
                            () -> searcher.postProcessBySort(hits, invalidThirdField, null, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("fieldName is required for sort field at index 2");
        }

        @Test
        void shouldThrowExceptionForInvalidSortDepth() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();
            String validSortJson =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"}]";

            // Test sortBySortDepth = 0
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, validSortJson, 0, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBySortDepth must be greater than or equal to 1. Found: 0");

            // Test sortBySortDepth = -1
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, validSortJson, -1, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBySortDepth must be greater than or equal to 1. Found: -1");

            // Test sortBySortDepth = -10 (more negative)
            assertThatThrownBy(() -> searcher.postProcessBySort(hits, validSortJson, -10, 10, 0))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "sortBySortDepth must be greater than or equal to 1. Found: -10");
        }

        @Test
        void shouldAcceptValidSortDepth() {
            HybridSearcher searcher = new HybridSearcher();
            HitGroup hits = createDummyHitGroup();
            String validSortJson =
                    "[{\"field_name\":\"f1\",\"order\":\"asc\",\"missing\":\"last\"}]";

            // Test sortBySortDepth = null (should work - uses default)
            HitGroup result1 = searcher.postProcessBySort(hits, validSortJson, null, 10, 0);
            assertThat((Object) result1).isNotNull();
            assertThat(result1.asList()).hasSize(1);

            // Test sortBySortDepth = 1 (should work)
            HitGroup result2 = searcher.postProcessBySort(hits, validSortJson, 1, 10, 0);
            assertThat((Object) result2).isNotNull();
            assertThat(result2.asList()).hasSize(1);

            // Test sortBySortDepth = 5 (should work)
            HitGroup result3 = searcher.postProcessBySort(hits, validSortJson, 5, 10, 0);
            assertThat((Object) result3).isNotNull();
            assertThat(result3.asList()).hasSize(1);

            // Test sortBySortDepth = 100 (larger than hit count, should work)
            HitGroup result4 = searcher.postProcessBySort(hits, validSortJson, 100, 10, 0);
            assertThat((Object) result4).isNotNull();
            assertThat(result4.asList()).hasSize(1);
        }
    }

    @Nested
    class UpdateQueryHitsOffsetsAndTargetHitsTest {

        @Test
        void shouldReturnOriginalQueryWhenNeitherFeatureEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query originalQuery = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            originalQuery, 100, 200, false, false);

            assertThat(result).isSameAs(originalQuery);
            assertThat(result.getHits()).isEqualTo(10);
            assertThat(result.getOffset()).isEqualTo(5);
        }

        @Test
        void shouldThrowExceptionWhenBothCandidatesAreNull() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");

            assertThatThrownBy(
                            () ->
                                    searcher.updateQueryHitsOffsetsAndTargetHits(
                                            query, null, null, true, false))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "Either relevantCandidates or sortByMinSortCandidates must be"
                                    + " provided");
        }

        @Test
        void shouldUpdateHitsWhenOnlyRelevanceCutoffEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 20, null, true, false);

            // Should use Math.min(relevantCandidates, limit+offset) = Math.min(20, 15) = 15
            assertThat(result.getHits()).isEqualTo(15);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateHitsWhenRelevantCandidatesLowerThanLimitOffset() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 8, null, true, false);

            // Should use Math.min(relevantCandidates, limit+offset) = Math.min(8, 15) = 8
            assertThat(result.getHits()).isEqualTo(8);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateHitsWhenOnlySortByEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, null, 20, false, true);

            // Should use Math.max(sortByMinSortCandidates, limit+offset) = Math.max(20, 15) = 20
            assertThat(result.getHits()).isEqualTo(20);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateHitsWhenSortByCandidatesLargerThanLimitOffset() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, null, 30, false, true);

            // Should use Math.max(sortByMinSortCandidates, limit+offset) = Math.max(30, 15) = 15
            assertThat(result.getHits()).isEqualTo(30);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldNotUpdateHitsWhenSortByCandidatesLowerThanLimitOffset() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, null, 8, false, true);

            // Should use Math.max(sortByMinSortCandidates, limit+offset) = Math.max(8, 15) = 15
            assertThat(result.getHits()).isEqualTo(15);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateHitsWhenBothFeaturesEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 30, 25, true, true);

            // Should use Math.max(relevantCandidates, sortByMinSortCandidates) = Math.max(30, 25)
            // = 30
            assertThat(result.getHits()).isEqualTo(30);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateHitsWhenBothFeaturesEnabledReversed() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 25, 30, true, true);

            // Should use Math.max(relevantCandidates, sortByMinSortCandidates) = Math.max(25, 30)
            // = 30
            assertThat(result.getHits()).isEqualTo(30);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldUpdateTensorTargetHitsWhenRelevanceCutoffEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950}");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 20, null, true, false);

            // Should use Math.min(newHits, currentTensorTargetHits) = Math.min(15, 50) = 15
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql).contains("targetHits: 15");
            assertThat(updatedYql).doesNotContain("targetHits: 50");
            assertThat(updatedYql).contains("hnsw.exploreAdditionalHits: 1985");
            assertThat(updatedYql).doesNotContain("hnsw.exploreAdditionalHits: 1950");
        }

        @Test
        void shouldUpdateTensorTargetHitsWhenSortByEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950}");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, null, 60, false, true);

            // Should use Math.max(newHits, currentTensorTargetHits) = Math.max(60, 50) = 60
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql).contains("targetHits: 60");
            assertThat(updatedYql).doesNotContain("targetHits: 50");
            assertThat(updatedYql).contains("hnsw.exploreAdditionalHits: 1940");
            assertThat(updatedYql).doesNotContain("hnsw.exploreAdditionalHits: 1950");
        }

        @Test
        void shouldUpdateTensorTargetHitsWhenBothFeaturesEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 40,"
                                    + " hnsw.exploreAdditionalHits: 1960}");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 30, 35, true, true);

            // newHits = Math.max(30, 35) = 35
            // newTensorTargetHits = Math.max(35, 40) = 40
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql).contains("targetHits: 40");
            assertThat(updatedYql).contains("hnsw.exploreAdditionalHits: 1960");
            assertThat(result.getHits()).isEqualTo(35);
        }

        @Test
        void shouldSetTensorTargetHitsToNewHitsWhenBothFeaturesEnabledAndOverrideEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where ({targetHits: 20,"
                                    + " hnsw.exploreAdditionalHits: 1980}nearestNeighbor(f, q))");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 30, 25, true, true, false, true, false);

            // newHits = Math.max(relevantCandidates=30, sortByMinSortCandidates=25) = 30
            // overrideLimitPlusOffset=true: newTensorTargetHits = newHits = 30 (not max with 20)
            assertThat(result.getHits()).isEqualTo(30);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where ({targetHits: 30,"
                                    + " hnsw.exploreAdditionalHits: 1970}nearestNeighbor(f, q))");
        }

        @Test
        void shouldReduceTensorTargetHitsWhenBothFeaturesEnabledAndOverrideEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where ({targetHits: 500,"
                                    + " hnsw.exploreAdditionalHits: 1500}nearestNeighbor(f, q))");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 30, 25, true, true, false, true, false);

            // newHits = Math.max(relevantCandidates=30, sortByMinSortCandidates=25) = 30
            // overrideLimitPlusOffset=true: newTensorTargetHits = newHits = 30
            // (NOT Math.max(30, 500) = 500, which would be the old behaviour)
            assertThat(result.getHits()).isEqualTo(30);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where ({targetHits: 30,"
                                    + " hnsw.exploreAdditionalHits: 1970}nearestNeighbor(f, q))");
        }

        @Test
        void shouldThrowExceptionWhenTensorTargetHitsLowerThanLimitOffset() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 10,"
                                    + " hnsw.exploreAdditionalHits: 1990}");

            assertThatThrownBy(
                            () ->
                                    searcher.updateQueryHitsOffsetsAndTargetHits(
                                            query, 20, null, true, false))
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining(
                            "The targetHits in the tensor query should not be smaller than"
                                    + " limit+offset");
        }

        @Test
        void shouldHandleEmptyTensorYqlCorrectly() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            // Empty tensor YQL should be handled gracefully
            query.properties().set("marqo__yql.tensor", "");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 20, null, true, false);

            assertThat(result.getHits()).isEqualTo(15);
            assertThat(result.getOffset()).isEqualTo(0);
            // Empty YQL should remain empty
            assertThat(result.properties().getString("marqo__yql.tensor")).isEmpty();
        }

        @Test
        void shouldPreserveTensorYqlStructureWhenUpdatingTargetHits() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {param1: 'value', targetHits: 100,"
                                    + " hnsw.exploreAdditionalHits: 1900, param2: true}");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, null, 150, false, true);

            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where {param1: 'value', targetHits: 150,"
                                    + " hnsw.exploreAdditionalHits: 1850, param2: true}");
        }
    }

    @Nested
    class IntegrationRelevanceCutoffAndSortByTest {
        @Test
        void shouldHandleTensorTargetHitsWithBothFeaturesEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=5&offset=2");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 60,"
                                    + " hnsw.exploreAdditionalHits: 1940}");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 40, 45, true, true);

            // newHits = Math.max(40, 45) = 45
            // newTensorTargetHits = Math.max(45, 60) = 60 (keeps existing higher value)
            assertThat(result.getHits()).isEqualTo(45);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 60,"
                                    + " hnsw.exploreAdditionalHits: 1940}");
        }

        @Test
        void shouldUpdateTensorTargetHitsWhenNewValueHigher() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=5&offset=2");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 30,"
                                    + " hnsw.exploreAdditionalHits: 1970}");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 40, 45, true, true);

            // newHits = Math.max(40, 45) = 45
            // newTensorTargetHits = Math.max(45, 30) = 45 (uses new higher value)
            assertThat(result.getHits()).isEqualTo(45);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 45, "
                                    + "hnsw.exploreAdditionalHits: 1955}");
        }

        @Test
        void shouldHandleEqualCandidatesInBothFeatures() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=5&offset=2");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 40, 40, true, true);

            // Should use Math.max(40, 40) = 40
            assertThat(result.getHits()).isEqualTo(40);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldHandleComplexTensorYqlWithBothFeatures() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=5&offset=2");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {queryVector: [1,2,3], targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950, threshold: 0.8}");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 35, 40, true, true);

            // newHits = Math.max(35, 40) = 40
            // newTensorTargetHits = Math.max(40, 50) = 50
            assertThat(result.getHits()).isEqualTo(40);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where {queryVector: [1,2,3], targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950, threshold: 0.8}");
        }

        @Test
        void shouldHandleBoundaryConditionsWithBothFeatures() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=1&offset=0");

            Query result = searcher.updateQueryHitsOffsetsAndTargetHits(query, 1, 1, true, true);

            // Should use Math.max(1, 1) = 1
            assertThat(result.getHits()).isEqualTo(1);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldHandleLargeCandidateValuesWithBothFeatures() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 1000,"
                                    + " hnsw.exploreAdditionalHits: 1000}");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 500, 750, true, true);

            // newHits = Math.max(500, 750) = 750
            // newTensorTargetHits = Math.max(750, 1000) = 1000
            assertThat(result.getHits()).isEqualTo(750);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 1000,"
                                    + " hnsw.exploreAdditionalHits: 1000}");
        }

        @Test
        void shouldConvertZeroTargetHitsToOneButKeepHitsZero() {
            Query query = new Query("search/?query=test");
            query.setHits(50);
            query.setOffset(0);
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950}");

            HybridSearcher searcher = new HybridSearcher();

            // Call with relevantCandidates = 0, which should result in hits = 0 but targetHits = 1
            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(query, 0, null, true, false);

            // Verify hits is set to 0 (original relevantCandidates value)
            assertThat(result.getHits()).isEqualTo(0);

            // Verify targetHits was converted from 0 to 1 in the tensor YQL
            String updatedTensorYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedTensorYql)
                    .isEqualTo(
                            "select * from sources * where {targetHits: 1,"
                                    + " hnsw.exploreAdditionalHits: 1999}");
        }

        @Test
        void shouldHandleRelevanceCutoffLogicWithSortByPresent() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where {targetHits: 100,"
                                    + " hnsw.exploreAdditionalHits: 1900}");

            // When both are enabled, relevanceCutoff logic changes:
            // - Uses Math.max instead of Math.min for determining newHits
            // - Uses Math.max instead of Math.min for tensorTargetHits
            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query,
                            8,
                            12,
                            true,
                            true // relevantCandidates < limit+offset, but with sortBy enabled
                            );

            // Should use Math.max(8, 12) = 12 (not Math.min like pure relevance cutoff)
            assertThat(result.getHits()).isEqualTo(12);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql).contains("targetHits: 100"); // Math.max(12, 100) = 100
            assertThat(updatedYql).contains("hnsw.exploreAdditionalHits: 1900");
        }
    }

    @Nested
    class OverrideLimitPlusOffsetTest {

        @Test
        void shouldExpandHitsToRelevantCandidatesWhenOverrideEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 200, null, true, false, false, true, false);

            // overrideLimitPlusOffset=true: newHits = relevantCandidates = 200
            // (expands beyond limit+offset=10)
            assertThat(result.getHits()).isEqualTo(200);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldReduceHitsBelowLimitWhenRelevantCandidatesSmall() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 3, null, true, false, false, true, false);

            // overrideLimitPlusOffset=true: newHits = relevantCandidates = 3
            // (can also reduce below limit+offset)
            assertThat(result.getHits()).isEqualTo(3);
        }

        @Test
        void shouldExpandHitsWithNonZeroOffset() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=5");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 20, null, true, false, false, true, false);

            // overrideLimitPlusOffset=true: newHits = relevantCandidates = 20
            // (limit+offset=15, but we ignore that and use 20)
            assertThat(result.getHits()).isEqualTo(20);
            assertThat(result.getOffset()).isEqualTo(0);
        }

        @Test
        void shouldSetTensorTargetHitsToMaxWhenOverrideEnabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where ({targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950}nearestNeighbor(f, q))");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 200, null, true, false, false, true, false);

            // overrideLimitPlusOffset=true: newHits=200, newTensorTargetHits=relevantCandidates=200
            assertThat(result.getHits()).isEqualTo(200);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where ({targetHits: 200,"
                                    + " hnsw.exploreAdditionalHits: 1800}nearestNeighbor(f, q))");
        }

        @Test
        void shouldSetTensorTargetHitsToRelevantCandidatesEvenWhenExistingIsLarger() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where ({targetHits: 500,"
                                    + " hnsw.exploreAdditionalHits: 1500}nearestNeighbor(f, q))");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 100, null, true, false, false, true, false);

            // overrideLimitPlusOffset=true: newHits=100, newTensorTargetHits=relevantCandidates=100
            // (always set to relevantCandidates, even when existing tensor targetHits is larger)
            assertThat(result.getHits()).isEqualTo(100);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where ({targetHits: 100,"
                                    + " hnsw.exploreAdditionalHits: 1900}nearestNeighbor(f, q))");
        }

        @Test
        void shouldUseLimitPlusOffsetMinBehaviourWhenOverrideDisabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 200, null, true, false, false, false, false);

            // overrideLimitPlusOffset=false: newHits = min(200, 10) = 10
            assertThat(result.getHits()).isEqualTo(10);
        }

        @Test
        void shouldSetTensorTargetHitsToMinWhenOverrideDisabled() {
            HybridSearcher searcher = new HybridSearcher();
            Query query = new Query("?q=test&hits=10&offset=0");
            query.properties()
                    .set(
                            "marqo__yql.tensor",
                            "select * from sources * where ({targetHits: 50,"
                                    + " hnsw.exploreAdditionalHits: 1950}nearestNeighbor(f, q))");

            Query result =
                    searcher.updateQueryHitsOffsetsAndTargetHits(
                            query, 200, null, true, false, false, false, false);

            // overrideLimitPlusOffset=false: newHits=min(200,10)=10,
            // newTensorTargetHits=min(10,50)=10
            assertThat(result.getHits()).isEqualTo(10);
            String updatedYql = result.properties().getString("marqo__yql.tensor");
            assertThat(updatedYql)
                    .isEqualTo(
                            "select * from sources * where ({targetHits: 10,"
                                    + " hnsw.exploreAdditionalHits: 1990}nearestNeighbor(f, q))");
        }
    }

    @Nested
    class MarqoMetadataFieldsTest {

        @Test
        void shouldExcludeNullValuesFromJsonSerialization() {
            // Realistic case: sortBy disabled (sortCandidates=null) and cutoff disabled
            // (probeCandidates=null, relevantCandidates=null); postProcessCandidates is always set
            HybridSearcher.MarqoMetadataFields metadataWithNulls =
                    new HybridSearcher.MarqoMetadataFields(null, null, null, 10);

            StringBuilder json = new StringBuilder();
            metadataWithNulls.writeJson(json);
            String jsonString = json.toString();

            assertThat(jsonString).contains("\"postProcessCandidates\":10");
            assertThat(jsonString).doesNotContain("sortCandidates");
            assertThat(jsonString).doesNotContain("probeCandidates");
            assertThat(jsonString).doesNotContain("relevantCandidates");
            assertThat(jsonString).doesNotContain("null");
        }

        @Test
        void shouldIncludeAllNonNullValuesInJsonSerialization() {
            // Create metadata with all non-null values
            HybridSearcher.MarqoMetadataFields metadataComplete =
                    new HybridSearcher.MarqoMetadataFields(8, 12, 6, 20);

            // Test JSON serialization includes all values
            StringBuilder json = new StringBuilder();
            metadataComplete.writeJson(json);
            String jsonString = json.toString();

            // Should contain all values
            assertThat(jsonString).contains("\"sortCandidates\":8");
            assertThat(jsonString).contains("\"probeCandidates\":12");
            assertThat(jsonString).contains("\"relevantCandidates\":6");
            assertThat(jsonString).contains("\"postProcessCandidates\":20");

            // Should not contain null
            assertThat(jsonString).doesNotContain("null");
        }
    }
}
